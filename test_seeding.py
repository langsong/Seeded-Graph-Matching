import numpy as np
import time
from multiprocessing import Pool
import matplotlib.pyplot as plt
from Experiments import run_experiments
from graspologic.match import graph_match
from ExpandWhenStuck import graph_match_percolation
from config import *
from utils.Plotting import plot_results, format_for_plotting
from utils.SeedingMethods import *
from utils.Graphs import *
from utils.SeedQuality import *
def graspologic_algorithm(G1, G2, partial_match):

    _, perm_inds, _, _ = graph_match(
        G1,
        G2,
        partial_match=partial_match
    )

    return perm_inds

def match_ratio(predicted_permutation, optimal_permutation):
    """
    Computes the fraction of vertices correctly matched.
    """
    return np.mean(predicted_permutation == optimal_permutation)

# Plots results of 2+ seeding methods on the same algorithm and graph generation function over 25 trials of varying seed numbers
def compare_seeding(graph_gen_func, seeding_funcs_list, algorithm, seed_nums_list, n_trials=TRIALS_PER_SEED_NUMBER):
    results = run_experiments(
        graph_gen_funcs=[graph_gen_func],
        seeding_funcs=seeding_funcs_list,
        seed_nums=seed_nums_list,
        algorithms=[algorithm],
        n_trials=n_trials,
    )
    return results


def compare_algorithms(graph_gen_func, seeding_func, algorithms, seed_nums_list, n_trials=TRIALS_PER_SEED_NUMBER):
    results = run_experiments(
        graph_gen_funcs=[graph_gen_func],
        seeding_funcs=[seeding_func],
        seed_nums=seed_nums_list,
        algorithms=algorithms,
        n_trials=n_trials,
    )
    return results


def compare_seeding_sequential(graph_gen_func, seeding_funcs_list, algorithm, seed_nums_list, n_trials=TRIALS_PER_SEED_NUMBER):
    """
    Runs graph matching trials, extracts confidence intervals
    """
    # Initialize a dictionary to hold all results
    results = {func.__name__: {s: [] for s in seed_nums_list} for func in seeding_funcs_list}
    
    for s in seed_nums_list:
        print(f"Running trials for {s} seeds...")
        
        for _ in range(n_trials):
            # 1. Generate Graphs
            G1, G2_shuffled, optimal_perm = graph_gen_func()
            
            for seeding_func in seeding_funcs_list:
                # 2. Grab Seeds
                seeds_G1, seeds_G2 = seeding_func(G1, G2_shuffled, s, optimal_perm)
                partial_match = np.column_stack((seeds_G1, seeds_G2))
                
                # 3. Fit Graph Match Model
                perm_inds = algorithm(G1, G2_shuffled, partial_match)                
                # 4. Compute and Store Score
                score = match_ratio(perm_inds, optimal_perm)
                results[seeding_func.__name__][s].append(score)
    return results

def compare_algorithms_sequential(graph_gen_func, seeding_func, algorithms, seed_nums_list, n_trials=TRIALS_PER_SEED_NUMBER):
    """
    Compares graph matching algorithms
    using identical random initial seeds.
    """
    results = {
        alg.__name__: {s: [] for s in seed_nums_list}
        for alg in algorithms
    }

    for s in seed_nums_list:

        print(f"Running trials for {s} seeds...")

        for trial in range(n_trials):
            # Generate graphs
            G1, G2_shuffled, optimal_perm = graph_gen_func()

            # Generate same seeds for both algorithms
            seeds_G1, seeds_G2 = seeding_func(G1, G2_shuffled, s, optimal_perm)
            partial_match = np.column_stack((seeds_G1, seeds_G2))

            for alg in algorithms:
                perm_inds = alg(G1, G2_shuffled, partial_match)
                score = match_ratio(perm_inds, optimal_perm)

                results[alg.__name__][s].append(score)
            
    return results

def seed_quality_experiment(
    metric,
    n_trials=200,
    n_seeds=10,
    graph_generator=gen_ER_graphs
):
    """
    Runs repeated graph matching experiments using random seeds and
    plots neighborhood coverage versus final match ratio.

    Parameters
    ----------
    n_trials : int
        Number of trials to run.

    n_seeds : int
        Number of random seeds to use in each trial.

    graph_generator : function
        Function that returns (G1, G2, optimal_permutation).

    Returns
    -------
    coverage_values : list
        Neighborhood coverage for each trial.

    accuracy_values : list
        Match ratio for each trial.
    """

    coverage_values = []
    accuracy_values = []

    for trial in range(n_trials):

        # Generate graphs
        G1, G2, optimal_permutation = graph_generator()

        # Generate random seeds
        seeds_G1, seeds_G2 = random_seeds(
            G1,
            G2,
            n_seeds,
            optimal_permutation
        )

        # Compute neighborhood coverage
        coverage = metric(G1, G2, seeds_G1, seeds_G2, optimal_permutation)

        # Run graph matching
        predicted_perm = graph_match_percolation(
            G1,
            G2,
            np.column_stack((seeds_G1, seeds_G2))
        )

        # Compute match ratio
        accuracy = np.mean(predicted_perm == optimal_permutation)

        coverage_values.append(coverage)
        accuracy_values.append(accuracy)

        print(f"Trial {trial + 1}/{n_trials} complete")

    # Compute correlation
    correlation = np.corrcoef(coverage_values, accuracy_values)[0, 1]
    print(f"\nPearson correlation: {correlation:.4f}")

    # Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(coverage_values, accuracy_values)

    plt.xlabel("Number of non-seed vertices with at least r seed neighbors.")
    plt.ylabel("Match Ratio")
    plt.title(
        f"Step 1 Nodes ({n_trials} Trials, {n_seeds} Random Seeds)"
    )
    plt.grid(True)

    plt.show()

    return coverage_values, accuracy_values




if __name__ == "__main__":
    algorithms = [
        graspologic_algorithm,
        graph_match_percolation
    ]
    start = time.process_time()
    accuracies, runtimes = compare_seeding_sequential(graph_gen_func=gen_PAPER_graphs,
                    seeding_funcs_list=[random_seeds, betweenness_seeds, triangle_degree_ratio_seeds, highest_degree_seeds],
                    algorithm=graspologic_algorithm,
                    seed_nums_list=SEED_COUNTS,
                    n_trials=TRIALS_PER_SEED_NUMBER)
    
    t = time.process_time() - start
    print(f"{t:.2f} seconds")
    
    #formatted_acuracies = format_for_plotting(accuracies, "seeding_func")
    #formatted_runtimes = format_for_plotting(runtimes, "seeding_func")
    
    plot_results(accuracies, SEED_COUNTS, y_label="Match Ratio")
    plot_results(runtimes, SEED_COUNTS, y_label="CPU time")
    
    # print(res)


    # algorithms = [
    #     graspologic_algorithm,
    #     graph_match_percolation
    # ]
    # print("Starting benchmark...\n")


    # # -----------------------------
    # # Sequential version
    # # -----------------------------

    # print("Running sequential version...")

    # start = time.perf_counter()

    # sequential_results = compare_algorithms_sequential(
    #     graph_gen_func=gen_SBM_graphs,
    #     seeding_func=random_seeds,
    #     algorithms=algorithms,
    #     seed_nums_list=SEED_COUNTS,
    #     n_trials=TRIALS_PER_SEED_NUMBER
    # )

    # sequential_time = time.perf_counter() - start


    # print("\nSequential runtime:")
    # print(f"{sequential_time:.2f} seconds")


    # print("\nSequential results:")
    # print(sequential_results)



    # # -----------------------------
    # # Parallel version
    # # -----------------------------

    # print("\n\nRunning parallel version...")


    # start = time.perf_counter()

    # parallel_results = run_experiments(
    #     graph_gen_funcs=[gen_SBM_graphs],
    #     seeding_funcs=[random_seeds],
    #     seed_nums=SEED_COUNTS,
    #     algorithms=algorithms,
    #     n_trials=TRIALS_PER_SEED_NUMBER,
    # )
    # parallel_results = format_for_plotting(parallel_results, "algorithm")
    # parallel_time = time.perf_counter() - start


    # print("\nParallel runtime:")
    # print(f"{parallel_time:.2f} seconds")


    # print("\nParallel results:")
    # print(parallel_results)



    # # -----------------------------
    # # Speedup
    # # -----------------------------

    # print("\n\nSpeedup:")
    # print(
    #     f"{sequential_time / parallel_time:.2f}x faster"
    # )


    # Optional plotting
    # plot_results(
    #     parallel_results,
    #     SEED_COUNTS
    # )
    
    