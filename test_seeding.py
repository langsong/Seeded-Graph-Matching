import numpy as np
import time
from Experiments import run_experiments
from graspologic.match import graph_match
from ExpandWhenStuck import graph_match_percolation
from config import *
from utils.Plotting import *
from utils.SeedingMethods import *
from utils.Graphs import *

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

    results = format_for_plotting(results, "seeding_func")
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
    accuracies = {func.__name__: {s: [] for s in seed_nums_list} for func in seeding_funcs_list}
    runtimes = {func.__name__: {s: [] for s in seed_nums_list} for func in seeding_funcs_list}
    
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
                start = time.process_time()
                perm_inds = algorithm(G1, G2_shuffled, partial_match)
                end = time.process_time()
                runtime = start - end
                
                # 4. Compute and Store Score
                score = match_ratio(perm_inds, optimal_perm)

                # 5. Store results
                accuracies[seeding_func.__name__][s].append(score)
                runtimes[seeding_func.__name__][s].append(runtime)
    
    return accuracies, runtimes

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

        for _ in range(n_trials):
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
    