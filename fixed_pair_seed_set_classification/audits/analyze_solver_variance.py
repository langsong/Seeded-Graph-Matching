"""Summarize the saved five-run Pareto–Chung–Lu solver-variance audit."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
AUDIT_DIR = DATA_DIR / "solver_variance"
RUN_CSV = AUDIT_DIR / "paper_solver_variance_runs.csv"
SOURCE_CSV = DATA_DIR / "paper_seed_sets.csv"
SOURCE_METADATA = DATA_DIR / "paper_seed_sets.metadata.json"
AUDIT_METADATA = AUDIT_DIR / "paper_solver_variance.metadata.json"
REPORT_JSON = AUDIT_DIR / "paper_solver_variance_summary.json"


def variance_summary(values: pd.DataFrame) -> dict[str, float]:
    within = float(values.var(axis=1, ddof=1).mean())
    observed_between = float(values.mean(axis=1).var(ddof=1))
    between = max(0.0, observed_between - within / values.shape[1])
    return {
        "within_set_variance": within,
        "estimated_between_set_variance": between,
        "between_to_within_ratio": between / within if within else np.inf,
        "estimated_single_run_reliability": between / (between + within),
        "estimated_five_run_mean_reliability": between
        / (between + within / values.shape[1]),
    }


def cv_predictions(model, X, train_target, folds, classification):
    predictions = np.full(len(X), np.nan)
    for train_index, test_index in folds:
        fitted = clone(model).fit(X.iloc[train_index], train_target.iloc[train_index])
        predictions[test_index] = (
            fitted.predict_proba(X.iloc[test_index])[:, 1]
            if classification
            else fitted.predict(X.iloc[test_index])
        )
    return predictions


def main() -> None:
    runs = pd.read_csv(RUN_CSV)
    source = pd.read_csv(SOURCE_CSV).set_index("candidate_id")
    feature_columns = json.loads(SOURCE_METADATA.read_text())["feature_columns"]
    audit_metadata = json.loads(AUDIT_METADATA.read_text())
    solver_seeds = audit_metadata["solver_seeds"]
    n_sets = len(audit_metadata["candidate_ids"])
    if len(runs) != n_sets * len(solver_seeds):
        raise ValueError("The audit is incomplete.")
    if runs.duplicated(["candidate_id", "solver_seed"]).any():
        raise ValueError("The audit has duplicate runs.")

    accuracy = runs.pivot(
        index="candidate_id", columns="solver_seed", values="final_unseeded_accuracy"
    )[solver_seeds]
    h0 = runs.pivot(index="candidate_id", columns="solver_seed", values="h0_accuracy")[
        solver_seeds
    ]
    success = accuracy.ge(0.9)
    probability = success.mean(axis=1)
    majority = success.sum(axis=1).ge(3).astype(int)
    original = source.loc[accuracy.index]
    original_label = original["final_unseeded_accuracy"].ge(0.9).astype(int)
    original_difference = (
        accuracy[solver_seeds[0]] - original["final_unseeded_accuracy"]
    ).abs()

    split_reliabilities = []
    for left in itertools.combinations(solver_seeds, 2):
        right = [seed for seed in solver_seeds if seed not in left]
        split_reliabilities.append(
            spearmanr(accuracy[list(left)].mean(axis=1), accuracy[right].mean(axis=1))
            .statistic
        )

    best = runs.loc[runs.groupby("candidate_id")["sgm_objective_score"].idxmax()]
    best = best.set_index("candidate_id").loc[accuracy.index]
    any_success = success.any(axis=1)
    best_success = best["final_unseeded_accuracy"].ge(0.9)
    pairwise_label_disagreement = [
        float((success[left] != success[right]).mean())
        for left, right in itertools.combinations(solver_seeds, 2)
    ]

    summary = {
        "n_seed_sets": int(n_sets),
        "n_solver_runs": int(len(runs)),
        "original_run_max_absolute_difference": float(original_difference.max()),
        "final_accuracy_variance": variance_summary(accuracy),
        "h0_accuracy_variance": variance_summary(h0),
        "mean_within_set_final_accuracy_std": float(
            accuracy.std(axis=1, ddof=1).mean()
        ),
        "median_within_set_final_accuracy_range": float(
            (accuracy.max(axis=1) - accuracy.min(axis=1)).median()
        ),
        "fraction_sets_range_over_0_25": float(
            ((accuracy.max(axis=1) - accuracy.min(axis=1)) > 0.25).mean()
        ),
        "split_2_vs_3_spearman_mean": float(np.mean(split_reliabilities)),
        "split_2_vs_3_spearman_min": float(np.min(split_reliabilities)),
        "split_2_vs_3_spearman_max": float(np.max(split_reliabilities)),
        "success_count_distribution": {
            str(k): int((success.sum(axis=1) == k).sum()) for k in range(6)
        },
        "fraction_sets_with_mixed_success_outcomes": float(
            ((success.sum(axis=1) > 0) & (success.sum(axis=1) < 5)).mean()
        ),
        "mean_pairwise_solver_label_disagreement": float(
            np.mean(pairwise_label_disagreement)
        ),
        "original_single_run_success_rate": float(original_label.mean()),
        "five_run_mean_success_probability": float(probability.mean()),
        "majority_success_rate": float(majority.mean()),
        "original_vs_majority_label_disagreement": float(
            (original_label != majority).mean()
        ),
        "original_vs_five_run_mean_accuracy_spearman": float(
            spearmanr(original["final_unseeded_accuracy"], accuracy.mean(axis=1))
            .statistic
        ),
        "original_mean_final_accuracy": float(
            original["final_unseeded_accuracy"].mean()
        ),
        "five_run_mean_final_accuracy": float(accuracy.values.mean()),
        "best_objective_of_five_mean_final_accuracy": float(
            best["final_unseeded_accuracy"].mean()
        ),
        "best_objective_of_five_success_rate": float(
            best_success.mean()
        ),
        "any_of_five_success_rate": float(any_success.mean()),
        "best_objective_captures_available_success": float(
            best_success.loc[any_success].mean()
        ),
        "mean_run_seconds": float(runs["sgm_runtime_seconds"].mean()),
        "total_sgm_seconds": float(runs["sgm_runtime_seconds"].sum()),
    }

    X = original[feature_columns].reset_index(drop=True)
    y_original_accuracy = original["final_unseeded_accuracy"].reset_index(drop=True)
    y_mean_accuracy = accuracy.mean(axis=1).reset_index(drop=True)
    y_original_binary = original_label.reset_index(drop=True)
    y_majority = majority.reset_index(drop=True)
    y_probability = probability.reset_index(drop=True)
    strategy = original["sampling_strategy"].reset_index(drop=True)
    stratification = strategy + "_" + y_majority.astype(str)
    folds = list(
        StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(
            X, stratification
        )
    )

    classifiers = {
        "Logistic regression": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
        ),
        "Random forest": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(
                n_estimators=400,
                min_samples_leaf=5,
                max_features="sqrt",
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ),
        ),
    }
    regressors = {
        "Ridge": make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=10)
        ),
        "Random forest": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestRegressor(
                n_estimators=400,
                min_samples_leaf=5,
                max_features="sqrt",
                random_state=42,
                n_jobs=-1,
            ),
        ),
    }

    model_metrics = []
    for name, model in classifiers.items():
        for label, train_target in (
            ("original_single_run", y_original_binary),
            ("majority_of_five", y_majority),
        ):
            predicted = cv_predictions(model, X, train_target, folds, True)
            model_metrics.append(
                {
                    "task": "classification",
                    "model": name,
                    "training_label": label,
                    "auc_vs_majority": float(roc_auc_score(y_majority, predicted)),
                    "average_precision_vs_majority": float(
                        average_precision_score(y_majority, predicted)
                    ),
                    "mse_vs_empirical_success_probability": float(
                        np.mean((predicted - y_probability) ** 2)
                    ),
                }
            )
    for name, model in regressors.items():
        for label, train_target in (
            ("original_single_run", y_original_accuracy),
            ("mean_of_five", y_mean_accuracy),
        ):
            predicted = cv_predictions(model, X, train_target, folds, False)
            model_metrics.append(
                {
                    "task": "regression",
                    "model": name,
                    "training_label": label,
                    "mae_vs_five_run_mean": float(
                        mean_absolute_error(y_mean_accuracy, predicted)
                    ),
                    "r2_vs_five_run_mean": float(r2_score(y_mean_accuracy, predicted)),
                    "spearman_vs_five_run_mean": float(
                        spearmanr(y_mean_accuracy, predicted).statistic
                    ),
                }
            )
    summary["model_cv"] = model_metrics
    summary["top_feature_correlations_with_five_run_mean_accuracy"] = [
        {"feature": feature, "spearman": float(correlation)}
        for feature, correlation in sorted(
            (
                (feature, spearmanr(original[feature], accuracy.mean(axis=1)).statistic)
                for feature in feature_columns
                if original[feature].nunique(dropna=True) > 1
            ),
            key=lambda item: abs(item[1]),
            reverse=True,
        )[:6]
    ]
    summary["strategy_summary"] = {}
    for name in ("random", "feature_diverse"):
        ids = original.index[original["sampling_strategy"].eq(name)]
        summary["strategy_summary"][name] = {
            "n_sets": int(len(ids)),
            "original_success_rate": float(original_label.loc[ids].mean()),
            "mean_success_probability": float(probability.loc[ids].mean()),
            "mean_final_accuracy": float(accuracy.loc[ids].values.mean()),
            "mean_within_set_final_accuracy_std": float(
                accuracy.loc[ids].std(axis=1, ddof=1).mean()
            ),
            "label_disagreement": float(
                (original_label.loc[ids] != majority.loc[ids]).mean()
            ),
        }

    REPORT_JSON.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
