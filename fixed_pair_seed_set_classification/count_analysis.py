"""Helpers for seed_set_classification.ipynb.

Rebuild per-seed-set outcomes from the saved SGM runs, and fit grouped binomial
(logistic) models of the single-run success probability p(S). Only reads saved
collections; uses NumPy, SciPy, pandas and scikit-learn.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import gammaln
from scipy.stats import binomtest, rankdata, spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------- outcomes

def load_collection(data_dir, graph_model, success_threshold=0.90, restarts=None):
    """Return (data, metadata): one row per seed set with outcomes rebuilt from the runs.

    Added columns: success_count, run_count, success_fraction, mean_final_accuracy,
    and, for the first ``restarts`` runs (default: all), best_objective_accuracy,
    best_objective_success (objective ties go to the earliest run), any_run_success
    and objective_selection_miss (a success was available but not selected).
    """

    data_dir = Path(data_dir)
    paths = {name: data_dir / f"{graph_model}_seed_sets{suffix}" for name, suffix in
             [("data", ".csv"), ("runs", "_runs.csv"), ("metadata", ".metadata.json")]}
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(f"Required saved experiment file: {path}")
    metadata = json.loads(paths["metadata"].read_text())
    if metadata.get("status", "complete") != "complete":  # older collections have no status
        raise RuntimeError(f"Collection status is {metadata['status']!r}; finish it first.")
    data, runs = pd.read_csv(paths["data"]), pd.read_csv(paths["runs"])
    n_runs = int(metadata["solver_runs"])
    restarts = n_runs if restarts is None else restarts
    if not 1 <= restarts <= n_runs:
        raise ValueError(f"restarts must be between 1 and {n_runs}.")
    seed_keys = data["seed_vertices_graph1"].map(lambda value: tuple(sorted(json.loads(value))))
    checks = {
        "one row per candidate_id": not data["candidate_id"].duplicated().any(),
        "no duplicate seed sets": not seed_keys.duplicated().any(),
        "same candidates in both files": set(data["candidate_id"]) == set(runs["candidate_id"]),
        "distinct run ids per set": not runs.duplicated(["candidate_id", "sgm_run_id"]).any(),
        f"{n_runs} runs per set": runs.groupby("candidate_id").size().eq(n_runs).all(),
        "accuracies in [0, 1]": runs["final_unseeded_accuracy"].between(0, 1).all(),
        "finite objectives": np.isfinite(runs["sgm_objective_score"]).all(),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"Saved data failed checks: {failed}")

    runs = runs.sort_values(["candidate_id", "sgm_run_id"], kind="stable")
    runs["run_success"] = (runs["final_unseeded_accuracy"] >= success_threshold).astype(int)
    by_set = runs.groupby("candidate_id")
    first_runs = by_set.head(restarts)
    best = first_runs.loc[first_runs.groupby("candidate_id")["sgm_objective_score"].idxmax()]
    outcomes = pd.DataFrame({
        "success_count": by_set["run_success"].sum(),
        "run_count": by_set.size(),
        "mean_final_accuracy": by_set["final_unseeded_accuracy"].mean(),
        "best_objective_accuracy": best.set_index("candidate_id")["final_unseeded_accuracy"],
        "best_objective_success": best.set_index("candidate_id")["run_success"],
        "any_run_success": first_runs.groupby("candidate_id")["run_success"].max(),
    })
    data = data.drop(columns=[c for c in outcomes if c in data]).merge(
        outcomes, left_on="candidate_id", right_index=True)
    data["success_fraction"] = data["success_count"] / data["run_count"]
    data["objective_selection_miss"] = (
        data["any_run_success"].eq(1) & data["best_objective_success"].eq(0)).astype(int)
    data["group"] = np.where(data["sampling_strategy"].eq("swap_path"),
                             "swap: " + data["starting_rule"].astype(str), data["sampling_strategy"])
    return data, metadata


def noise_split(frame):
    """How much of one run's outcome is due to the seed set rather than solver randomness."""

    n = frame["run_count"].to_numpy(dtype=float)
    p = frame["success_fraction"].to_numpy(dtype=float)
    if len(p) < 2 or n.min() < 2:
        return {}
    within = np.mean(p * (1 - p) * n / (n - 1))
    between = max(np.var(p, ddof=1) - np.mean(within / n), 0.0)
    return {
        "mean_success_fraction": float(p.mean()),
        "mixed_outcome_share": float(np.mean((p > 0) & (p < 1))),
        "seed_set_share_of_one_run": float(between / (between + within)),
        "reliability_of_fraction": float(between / (between + within / n.mean())),
    }


# ---------------------------------------------------------------- features

def _spearman(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3 or np.unique(x[valid]).size < 2 or np.unique(y[valid]).size < 2:
        return np.nan
    return float(spearmanr(x[valid], y[valid]).statistic)


def _partial_spearman(frame, x, y, control):
    sample = frame[[x, y, control]].dropna()
    design = np.column_stack([np.ones(len(sample)), rankdata(sample[control])])

    def residual(values):
        ranks = rankdata(values)
        return ranks - design @ np.linalg.lstsq(design, ranks, rcond=None)[0]

    return float(np.corrcoef(residual(sample[x]), residual(sample[y]))[0, 1])


def feature_table(train, features, degree_feature):
    """Spearman correlation with the success fraction on random and on all training sets,
    and the partial correlation on random sets after controlling for degree."""

    random_train = train[train["sampling_strategy"].eq("random")]
    rows = [{
        "feature": column,
        "random_sets": _spearman(random_train[column], random_train["success_fraction"]),
        "random_sets_given_degree": (np.nan if column == degree_feature else
                                     _partial_spearman(random_train, column, "success_fraction", degree_feature)),
        "all_training_sets": _spearman(train[column], train["success_fraction"]),
    } for column in features]
    return pd.DataFrame(rows).set_index("feature").sort_values("random_sets", key=np.abs, ascending=False)


# ---------------------------------------------------------------- binomial model

def fit_binomial(frame, columns, C):
    """Logistic model of p(S) fitted to success counts (two weighted rows per set).

    This is the grouped-binomial likelihood up to a constant. A sample with only
    successes or only failures gets a smoothed intercept-only probability.
    """

    preprocessing = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    X = preprocessing.fit_transform(frame[columns].replace([np.inf, -np.inf], np.nan))
    successes = frame["success_count"].to_numpy(dtype=float)
    totals = frame["run_count"].to_numpy(dtype=float)
    fitted = {"columns": list(columns), "preprocessing": preprocessing, "model": None, "constant": None}
    if successes.sum() in (0, totals.sum()):
        fitted["constant"] = (successes.sum() + 0.5) / (totals.sum() + 1.0)
        return fitted
    weights = np.r_[successes, totals - successes]
    keep = weights > 0
    model = LogisticRegression(C=C, max_iter=5000)
    model.fit(np.vstack([X, X])[keep], np.r_[np.ones(len(frame)), np.zeros(len(frame))][keep],
              sample_weight=weights[keep])
    if model.n_iter_.max() >= model.max_iter:
        raise RuntimeError("Binomial fit reached its iteration limit.")
    fitted["model"] = model
    return fitted


def predict_binomial(fitted, frame):
    if fitted["constant"] is not None:
        return np.full(len(frame), fitted["constant"])
    X = fitted["preprocessing"].transform(frame[fitted["columns"]].replace([np.inf, -np.inf], np.nan))
    return fitted["model"].predict_proba(X)[:, 1]


def probability_metrics(frame, probabilities):
    """Per-run binomial NLL (with the count term) and Brier score, plus rank agreement."""

    p = np.clip(np.asarray(probabilities, dtype=float), 1e-12, 1 - 1e-12)
    y = frame["success_count"].to_numpy(dtype=float)
    n = frame["run_count"].to_numpy(dtype=float)
    log_choose = gammaln(n + 1) - gammaln(y + 1) - gammaln(n - y + 1)
    return {
        "binomial_nll_per_run": float(-(log_choose + y * np.log(p) + (n - y) * np.log1p(-p)).sum() / n.sum()),
        "brier_per_run": float((y * (1 - p) ** 2 + (n - y) * p ** 2).sum() / n.sum()),
        "spearman_success_fraction": _spearman(p, y / n),
        "mean_predicted_p": float(np.average(p, weights=n)),
        "observed_success_fraction": float(y.sum() / n.sum()),
    }


def select_models(random_train, other_train, feature_sets, c_grid, folds=5, repeats=10, seed=42):
    """Repeated K-fold over the random training sets (other training sets stay in every fit).

    Returns (all_results, chosen): metrics averaged over repeats for every model and C,
    and the C with the lowest out-of-fold NLL for each model.
    """

    rows = []
    for name, columns in feature_sets.items():
        for C in c_grid:
            per_repeat = []
            for repeat in range(repeats):
                predicted = np.full(len(random_train), np.nan)
                splitter = KFold(n_splits=folds, shuffle=True, random_state=seed + repeat)
                for fit_index, held_out in splitter.split(random_train):
                    fitted = fit_binomial(pd.concat([random_train.iloc[fit_index], other_train]), columns, C)
                    predicted[held_out] = predict_binomial(fitted, random_train.iloc[held_out])
                per_repeat.append(probability_metrics(random_train, predicted))
            rows.append({"model": name, "C": C, **pd.DataFrame(per_repeat).mean().to_dict()})
    results = pd.DataFrame(rows)
    chosen = results.loc[results.groupby("model", sort=False)["binomial_nll_per_run"].idxmin()]
    return results, chosen.set_index("model")


def evaluate_test(fit_sets, test, feature_sets, chosen_c, baseline_sets):
    """Fit each model with its chosen C, score it on the held-out test sets.

    ``baseline_sets`` gives the intercept-only reference (use the random training sets:
    constructed sets have a different success rate).
    """

    fitted = {name: fit_binomial(fit_sets, columns, chosen_c[name]) for name, columns in feature_sets.items()}
    probabilities = {name: predict_binomial(model, test) for name, model in fitted.items()}
    baseline = baseline_sets["success_count"].sum() / baseline_sets["run_count"].sum()
    rows = {name: probability_metrics(test, p) for name, p in probabilities.items()}
    rows["Random-training mean (no features)"] = probability_metrics(test, np.full(len(test), baseline))
    table = pd.DataFrame(rows).T
    reference = next(iter(feature_sets))
    for metric in ["binomial_nll_per_run", "brier_per_run"]:
        table[f"{metric}_vs_{reference}"] = table[metric] - table.loc[reference, metric]
    return table, probabilities, fitted


def calibration_table(test, probabilities, bins=4):
    rows = []
    for name, p in probabilities.items():
        bin_ids = (pd.qcut(p, q=min(bins, np.unique(p).size), labels=False, duplicates="drop")
                   if np.unique(p).size > 1 else np.zeros(len(p), dtype=int))
        sample = test[["success_count", "run_count"]].assign(predicted_p=p, bin=bin_ids)
        for bin_id, group in sample.groupby("bin"):
            rows.append({"model": name, "bin": int(bin_id), "sets": len(group),
                         "predicted_p": np.average(group["predicted_p"], weights=group["run_count"]),
                         "observed_fraction": group["success_count"].sum() / group["run_count"].sum()})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- restarts and selection

def restart_table(test, probabilities, restarts):
    """Observed restart outcomes next to the model's 1 - (1 - p)^R."""

    rows = {name: {"predicted_any_run_success": float(np.mean(1 - (1 - p) ** restarts))}
            for name, p in probabilities.items()}
    table = pd.DataFrame(rows).T
    table["observed_any_run_success"] = test["any_run_success"].mean()
    table["observed_best_objective_success"] = test["best_objective_success"].mean()
    table["objective_selection_misses"] = int(test["objective_selection_miss"].sum())
    return table


def selection_table(test, probabilities, fractions=(0.10, 0.25, 0.50), seed=42):
    """Outcomes of the top-ranked fraction of the test pool for each model (ties: fixed random order)."""

    tie_order = np.random.default_rng(seed).permutation(len(test))

    def summary(selected):
        return {"sets": len(selected),
                "single_run_success": selected["success_count"].sum() / selected["run_count"].sum(),
                "best_objective_success": selected["best_objective_success"].mean(),
                "mean_best_objective_accuracy": selected["best_objective_accuracy"].mean()}

    rows = [{"model": "All test sets", "top_fraction": 1.0, **summary(test)}]
    for name, p in probabilities.items():
        ranked = test.assign(_p=p, _tie=tie_order).sort_values(["_p", "_tie"], ascending=[False, True])
        for fraction in fractions:
            rows.append({"model": name, "top_fraction": fraction,
                         **summary(ranked.head(max(1, int(np.ceil(fraction * len(test))))))})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- constructed sets

def group_residuals(data, random_train, columns, C):
    """Fit on random training sets only, then compare observed and predicted success by group.

    A clearly negative residual means a group does worse than the random-set relationship
    predicts, i.e. that relationship does not extrapolate there.
    """

    fitted = fit_binomial(random_train, columns, C)
    frame = data.assign(predicted_p=predict_binomial(fitted, data))
    table = frame.groupby("group").agg(
        sets=("candidate_id", "size"), observed_fraction=("success_fraction", "mean"),
        predicted_p=("predicted_p", "mean"), best_objective_success=("best_objective_success", "mean"))
    table["residual"] = table["observed_fraction"] - table["predicted_p"]
    return table.sort_values("predicted_p")


def contrast_pairs(data, degree_feature):
    """One row per degree-controlled contrast pair (high minus low)."""

    contrasts = data[data["sampling_strategy"].eq("degree_controlled")].sort_values("candidate_id")
    if "contrast_pair_id" not in contrasts or contrasts["contrast_pair_id"].isna().all():
        # Sampler version 1 has no pair IDs: pairs are consecutive low/high rows.
        is_low = contrasts["contrast_tail"].eq("low").to_numpy()
        starts = is_low[:-1] & ~is_low[1:]
        pair_id = np.full(len(contrasts), np.nan)
        for number, start in enumerate(np.flatnonzero(starts)):
            pair_id[start:start + 2] = number
        contrasts = contrasts.assign(contrast_pair_id=pair_id).dropna(subset=["contrast_pair_id"])
    rows = []
    for pair_id, pair in contrasts.groupby("contrast_pair_id"):
        low, high = (pair[pair["contrast_tail"].eq(tail)].iloc[0] for tail in ("low", "high"))
        rows.append({
            "pair": int(pair_id), "feature": high["contrast_feature"],
            "delta_success_fraction": high["success_fraction"] - low["success_fraction"],
            "delta_best_objective_success": high["best_objective_success"] - low["best_objective_success"],
            "delta_degree": high[degree_feature] - low[degree_feature],
            "uninformative": low["success_count"] == high["success_count"]
            and high["success_count"] in (0, high["run_count"]),
        })
    return pd.DataFrame(rows)


def contrast_summary(pairs):
    """Per contrast feature: mean difference, wins/ties/losses and a sign test (ties excluded)."""

    def summarize(group):
        wins = int((group["delta_success_fraction"] > 0).sum())
        losses = int((group["delta_success_fraction"] < 0).sum())
        return pd.Series({
            "pairs": len(group),
            "mean_delta_success_fraction": group["delta_success_fraction"].mean(),
            "wins": wins, "ties": len(group) - wins - losses, "losses": losses,
            "sign_test_p": binomtest(wins, wins + losses).pvalue if wins + losses else np.nan,
            "mean_delta_best_objective_success": group["delta_best_objective_success"].mean(),
            "mean_delta_degree": group["delta_degree"].mean(),
            "uninformative_share": group["uninformative"].mean(),
        })

    return pd.DataFrame({feature: summarize(group) for feature, group in pairs.groupby("feature")}).T
