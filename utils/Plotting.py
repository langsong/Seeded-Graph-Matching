import numpy as np
from config import *
import matplotlib.pyplot as plt
import scipy.stats as st


def plot_results(results, seed_nums_list=SEED_COUNTS):
    """
    Plots mean accuracy and 95% confidence intervals.

    Parameters
    ----------
    results:
        {
            label:
                {
                    n_seeds:
                        [accuracy scores]
                }
        }

    seed_nums_list:
        list of seed counts tested
    """

    plt.figure(figsize=(10,6))

    for alg in results:

        means = []
        ci_lower = []
        ci_upper = []

        for s in seed_nums_list:

            data = results[alg][s]

            mean_val = np.mean(data)

            se = st.sem(data) if len(data) > 1 else 0
            ci = 1.96 * se

            means.append(mean_val)
            ci_lower.append(mean_val-ci)
            ci_upper.append(mean_val+ci)


        plt.plot(
            seed_nums_list,
            means,
            marker='o',
            label=alg
        )

        plt.fill_between(
            seed_nums_list,
            ci_lower,
            ci_upper,
            alpha=0.2
        )


    plt.xlabel("Number of Seeds")
    plt.ylabel("Match Ratio")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    
def format_for_plotting(grouped_results, label_key):
    """
    Converts grouped experiment results into the format expected by plot_results.

    Parameters
    ----------
    grouped_results : list
        Output of run_experiments()

    label_key : str
        Which field should become the plot legend.
        Examples:
            "algorithm"
            "seeding_func"
            "graph_gen_func"
    """

    results = {}

    for experiment in grouped_results:

        label = experiment[label_key].__name__
        seed_count = experiment["seed_count"]

        if label not in results:
            results[label] = {}

        results[label][seed_count] = experiment["scores"]

    return results

def plot_seed_quality(
    metric_values,
    accuracy_values,
    metric_name,
    xlabel,
    n_trials,
    n_seeds
):
    """
    Plots a seed metric against graph matching accuracy.
    """

    correlation = np.corrcoef(metric_values, accuracy_values)[0, 1]
    print(f"\nPearson correlation: {correlation:.4f}")

    plt.figure(figsize=(8, 6))
    plt.scatter(metric_values, accuracy_values)

    plt.xlabel(xlabel)
    plt.ylabel("Match Ratio")
    plt.title(
        f"{metric_name} ({n_trials} Trials, {n_seeds} Seeds)"
    )

    plt.grid(True)
    plt.show()


    import numpy as np
import matplotlib.pyplot as plt


def plot_match_ratio_histogram(results, 
                               graph_gen_func=None,
                               seeding_func=None,
                               algorithm=None,
                               seed_count=None,
                               bins=10):
    """
    Creates a histogram of match ratios from run_experiments() output.

    Parameters
    ----------
    results : list[dict]
        Output from run_experiments()

    graph_gen_func, seeding_func, algorithm, seed_count : optional
        Filters for selecting specific experiments.
        Leave None to include all.

    bins : int
        Number of match ratio categories.

    """

    # Collect all scores that match filters
    scores = []

    for experiment in results:

        if graph_gen_func is not None:
            if experiment["graph_gen_func"] != graph_gen_func:
                continue

        if seeding_func is not None:
            if experiment["seeding_func"] != seeding_func:
                continue

        if algorithm is not None:
            if experiment["algorithm"] != algorithm:
                continue

        if seed_count is not None:
            if experiment["seed_count"] != seed_count:
                continue

        scores.extend(experiment["scores"])


    if len(scores) == 0:
        raise ValueError("No scores matched the given filters.")

    
    # Create histogram bins
    bin_edges = np.linspace(0, 1, bins + 1)

    counts, edges = np.histogram(
        scores,
        bins=bin_edges
    )

    # Make labels like "0.0-0.1"
    labels = [
        f"{edges[i]:.1f}-{edges[i+1]:.1f}"
        for i in range(len(edges)-1)
    ]


    # Plot
    plt.figure(figsize=(10, 5))

    plt.bar(
        labels,
        counts
    )

    plt.xlabel("Match Ratio Range")
    plt.ylabel("Number of Trials")
    plt.title("Distribution of Graph Matching Performance")

    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()