import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st


def plot_results(results, seed_nums_list):
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