import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st


def plot_results(results, seed_nums_list, y_label="Match Ratio", show_plot=True, out_file=None):
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
    plt.ylabel(y_label)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    if out_file:
        plt.savefig(out_file)
    
    if show_plot:
        plt.show()

    
def format_for_plotting(grouped_results : list, label_key):
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