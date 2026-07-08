import numpy as np
from multiprocessing import Pool

def run_trial(graph_gen_func, seeding_func, n_seeds, algorithm):
    """
    Runs a single graph matching trial using one algorithm, one seeding method, one graph gen func
    Returns
    -------
    float
        Match accuracy for this trial.
    """
    # Generate graph pair
    G1, G2_shuffled, optimal_perm = graph_gen_func()

    # Generate seeds
    seeds_G1, seeds_G2 = seeding_func(G1, G2_shuffled, n_seeds, optimal_perm)
    partial_match = np.column_stack((seeds_G1, seeds_G2))

    # Generate prediction and return score
    predicted_perm = algorithm(G1, G2_shuffled, partial_match)
    score = np.mean(predicted_perm == optimal_perm)
    return score


def build_jobs(graph_gen_funcs, seeding_funcs, seed_nums, algorithms, n_trials):
    """
    Creates a list of independent graph matching jobs.
    """
    jobs = []

    for n_seeds in seed_nums:
        for _ in range(n_trials):
            for algorithm in algorithms:
                    for seeding_func in seeding_funcs:
                        for graph_gen_func in graph_gen_funcs:
                            jobs.append(
                                {
                                    "graph_gen_func": graph_gen_func,
                                    "seeding_func": seeding_func,
                                    "seed_count": n_seeds,
                                    "algorithm": algorithm,
                                }
                            )
    return jobs

def run_trial_wrapper(job):
    """
    Allows multiprocessing to call run_trial
    using a single argument.
    """
    return job, run_trial(
        job["graph_gen_func"],
        job["seeding_func"],
        job["seed_count"],
        job["algorithm"],
    )


def run_experiments(graph_gen_funcs, seeding_funcs, seed_nums, algorithms, n_trials):
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
    
    jobs = build_jobs(
        graph_gen_funcs,
        seeding_funcs,
        seed_nums,
        algorithms,
        n_trials,
    )
    # run jobs in parallel
    with Pool(processes=8) as pool:
        raw_results = []

        total_jobs = len(jobs)

        next_print = 10  # next percentage to print

        for i, result in enumerate(
            pool.imap_unordered(
                run_trial_wrapper,
                jobs
            ),
            start=1
        ):
            raw_results.append(result)

            percent_complete = (i / total_jobs) * 100

            if percent_complete >= next_print:
                print(
                    f"Completed {percent_complete:.0f}% "
                    f"({i}/{total_jobs} trials)"
                )
                next_print += 10


    # THIS IS FOR NO PROGRESS PRINT STATEMENTS
    # with Pool(processes=8) as pool:
    #     raw_results = pool.map(
    #         run_trial_wrapper,
    #         jobs
    #     )
    
        # Group identical experiments together
    grouped = {}

    for job, score in raw_results:

        key = (
            job["graph_gen_func"],
            job["seeding_func"],
            job["algorithm"],
            job["seed_count"],
        )

        if key not in grouped:
            grouped[key] = {
                "graph_gen_func": job["graph_gen_func"],
                "seeding_func": job["seeding_func"],
                "algorithm": job["algorithm"],
                "seed_count": job["seed_count"],
                "scores": [],
            }

        grouped[key]["scores"].append(score)

    # Convert dictionary to list
    grouped_results = list(grouped.values())

    return grouped_results



    # # Initialize result dictionary
    # seed_nums = sorted(
    #     set(job[2] for job in jobs)
    # )

    # algorithms = jobs[0][3]

    # results = {
    #     algorithm: {
    #         n_seeds: []
    #         for n_seeds in seed_nums
    #     }
    #     for algorithm in algorithms
    # }


    # # Organize results
    # for job, scores in raw_results:

    #     n_seeds = job[2]

    #     for algorithm_name, score in scores.items():

    #         results[algorithm_name][n_seeds].append(score)


    # return results

# def run_experiments(jobs):
#     """
#     Runs a list of graph matching jobs in parallel.

#     Parameters
#     ----------
#     jobs : list of dict

#     Returns
#     -------
#     results:
#         {
#             label:
#                 {
#                     seed_count:
#                         [scores]
#                 }
#         }
#     """

#     # Run jobs in parallel
#     with Pool(processes=8) as pool:
#         raw_results = pool.map(
#             run_trial_wrapper,
#             jobs
#         )

#     # Discover labels and seed counts
#     labels = sorted(
#         {job["label"] for job in jobs}
#     )

#     seed_counts = sorted(
#         {job["seed_count"] for job in jobs}
#     )

#     # Initialize results dictionary
#     results = {
#         label: {
#             seed_count: []
#             for seed_count in seed_counts
#         }
#         for label in labels
#     }

#     # Store scores
#     for job, score in raw_results:

#         label = job["label"]
#         seed_count = job["seed_count"]

#         results[label][seed_count].append(score)

#     return results