import numpy as np
from multiprocessing import Pool
from graspologic.match import graph_match
from ExpandWhenStuck import graph_match_percolation
from config import *
from Plotting import plot_results
from SeedingMethods import random_seeds, blocked_random_seeds, highest_degree_seeds, blocked_highest_degree_seeds, betweenness_seeds
from Graphs import gen_ER_graphs, gen_SBM_graphs

def match_ratio(predicted_permutation, optimal_permutation):
    """
    Computes the fraction of vertices correctly matched.
    """
    return np.mean(predicted_permutation == optimal_permutation)

def compare_seeding(graph_gen_func, seeding_funcs_list, seed_nums_list, n_trials=TRIALS_PER_SEED_NUMBER):
    """
    Runs graph matching trials, extracts confidence intervals, and plots the results.
    """
    # Initialize a dictionary to hold all results
    results = {func.__name__: {s: [] for s in seed_nums_list} for func in seeding_funcs_list}
    
    for s in seed_nums_list:
        print(f"Running trials for {s} seeds...")
        
        for trial in range(n_trials):
            # 1. Generate Graphs
            G1, G2_shuffled, optimal_perm = graph_gen_func()
            
            for seeding_func in seeding_funcs_list:
                # 2. Grab Seeds
                seeds_G1, seeds_G2 = seeding_func(G1, G2_shuffled, s, optimal_perm)
                partial_match = np.column_stack((seeds_G1, seeds_G2))
                
                # 3. Fit Graph Match Model
                _, perm_inds, _, _ = graph_match(G1, G2_shuffled, partial_match=partial_match)
                
                # 4. Compute and Store Score
                score = match_ratio(perm_inds, optimal_perm)
                results[seeding_func.__name__][s].append(score)
                
    # --- Plotting the Results ---
    plt.figure(figsize=(10, 6))
    
    for func_name in results:
        means = []
        ci_lower = []
        ci_upper = []
        
        for s in seed_nums_list:
            data = results[func_name][s]
            mean_val = np.mean(data)
            
            # Calculate 95% Confidence Interval using Standard Error
            se = st.sem(data) if len(data) > 1 else 0
            ci = se * 1.96 
            
            means.append(mean_val)
            ci_lower.append(mean_val - ci)
            ci_upper.append(mean_val + ci)
            
        # Plot mean lines
        plt.plot(seed_nums_list, means, label=func_name, marker='o')
        # Plot confidence intervals
        plt.fill_between(seed_nums_list, ci_lower, ci_upper, alpha=0.2)
        
    plt.xlabel('Number of Seeds', fontsize=12)
    plt.ylabel('Match Ratio (Accuracy)', fontsize=12)
    plt.title('Seeded Graph Matching: Random vs. Blocked Random Seeding', fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    
    return results


def compare_algorithms(graph_gen_func, seeding_func, seed_nums_list, n_trials=TRIALS_PER_SEED_NUMBER):
    """
    Compares graph matching algorithms
    using identical random initial seeds.
    """

    algorithms = ["graspologic_graph_match", "expand_when_stuck"]

    results = {
        alg: {s: [] for s in seed_nums_list}
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


            # Algorithm 1: Graspologic
            _, perm_inds, _, _ = graph_match(G1, G2_shuffled, partial_match=partial_match)

            score = match_ratio(perm_inds, optimal_perm)

            results["graspologic_graph_match"][s].append(score)

            # Algorithm 2: ExpandWhenStuck

            perm_inds = graph_match_percolation(
                G1,
                G2_shuffled,
                partial_match,
                r=2,
                ExpandWhenStuck=True
            )


            score = match_ratio(perm_inds, optimal_perm)
            results["expand_when_stuck"][s].append(score)


    # Plot results
    plt.figure(figsize=(10,6))

    for alg in results:

        means = []
        ci_lower = []
        ci_upper = []

        for s in seed_nums_list:

            data = results[alg][s]

            mean_val = np.mean(data)

            se = st.sem(data)
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


    return results

def graspologic_algorithm(G1, G2, partial_match):

    _, perm_inds, _, _ = graph_match(
        G1,
        G2,
        partial_match=partial_match
    )

    return perm_inds


def run_trial(graph_gen_func, seeding_func, n_seeds, algorithms):
    """
    Runs a single graph matching trial.
    Returns
    -------
    dictionary matching algorithms to their scores, eg.
        {
            "graspologic_graph_match": score,
            "expand_when_stuck": score,
            ...
        }
    """

    # Generate graph pair
    G1, G2_shuffled, optimal_perm = graph_gen_func()

    # Generate seeds
    seeds_G1, seeds_G2 = seeding_func(G1, G2_shuffled, n_seeds, optimal_perm)

    partial_match = np.column_stack((seeds_G1, seeds_G2))

    scores = {}

    for name, algorithm in algorithms.items():
        predicted_perm = algorithm(G1, G2_shuffled, partial_match)
        scores[name] = match_ratio(
            predicted_perm,
            optimal_perm
        )
    return scores


def build_jobs(
    graph_gen_func,
    seeding_func,
    seed_nums,
    algorithms,
    n_trials,
):
    """
    Creates a list of independent graph matching jobs.
    """

    jobs = []

    for n_seeds in seed_nums:
        for _ in range(n_trials):

            jobs.append(
                (
                    graph_gen_func,
                    seeding_func,
                    n_seeds,
                    algorithms,
                )
            )

    return jobs

def run_trial_wrapper(job):
    """
    Allows multiprocessing to call run_trial
    using a single argument.
    """
    return job, run_trial(*job)

def run_experiments(jobs):
    """
    Runs a list of graph matching jobs in parallel.

    Returns
    -------
    results:
        {
            algorithm_name:
                {
                    n_seeds:
                        [accuracy scores]
                }
        }
    """

    # Run jobs in parallel
    with Pool() as pool:
        raw_results = pool.map(
            run_trial_wrapper,
            jobs
        )


    # Initialize result dictionary
    seed_nums = sorted(
        set(job[2] for job in jobs)
    )

    algorithms = jobs[0][3]

    results = {
        algorithm: {
            n_seeds: []
            for n_seeds in seed_nums
        }
        for algorithm in algorithms
    }


    # Organize results
    for job, scores in raw_results:

        n_seeds = job[2]

        for algorithm_name, score in scores.items():

            results[algorithm_name][n_seeds].append(score)


    return results


if __name__ == "__main__":    
    
    # print("Initiating SGM Experiments... this might take a minute depending on core count.")
    # compare_algorithms(
    #     graph_gen_func=gen_SBM_graphs,
    #     seeding_func=random_seeds,
    #     seed_nums_list=SEED_COUNTS,
    #     n_trials=TRIALS_PER_SEED_NUMBER
    # )

    algorithms = {
        "graspologic_graph_match": graspologic_algorithm,
        "expand_when_stuck": graph_match_percolation,
    }

    jobs = build_jobs(
    graph_gen_func=gen_SBM_graphs,
    seeding_func=random_seeds,
    seed_nums=SEED_COUNTS,
    algorithms=algorithms,
    n_trials=TRIALS_PER_SEED_NUMBER,
)


    results = run_experiments(jobs)

    print(results)
    # compare_seeding(
    #     graph_gen_func=gen_SBM_graphs,
    #     seeding_funcs_list=[random_seeds, blocked_random_seeds, highest_degree_seeds],
    #     seed_nums_list=SEED_COUNTS,
    #     n_trials=TRIALS_PER_SEED_NUMBER
    # )
