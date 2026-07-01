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
            algorithm_name:
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


    plt.xlabel("Number of Random Seeds")
    plt.ylabel("Match Ratio")
    plt.title("Graph Matching Algorithm Comparison")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()