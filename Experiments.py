import numpy as np
from multiprocessing import Pool
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
        scores[name] = np.mean(predicted_perm == optimal_perm)
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
    with Pool(processes=8) as pool:
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