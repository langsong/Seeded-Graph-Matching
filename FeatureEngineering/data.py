import numpy as np
import networkx as nx
import pandas as pd
from multiprocessing import Pool
from Experiments import build_jobs



def generate_feature_dataset(graph_gen_funcs, seeding_funcs, seed_nums, algorithms, n_trials):
    """
    Generates a dataframe of seed set features and matching outcomes.

    Returns
    -------
    pandas.DataFrame
        One row per graph matching trial.
    """

    jobs = build_jobs(
        graph_gen_funcs,
        seeding_funcs,
        seed_nums,
        algorithms,
        n_trials,
    )

    with Pool(processes=8) as pool:

        dataset = []

        total_jobs = len(jobs)
        next_print = 10

        for i, row in enumerate(
            pool.imap_unordered(
                run_feature_trial_wrapper,
                jobs
            ),
            start=1
        ):

            dataset.append(row)

            percent_complete = (i / total_jobs) * 100

            if percent_complete >= next_print:
                print(
                    f"Generated {percent_complete:.0f}% "
                    f"({i}/{total_jobs} feature rows)"
                )
                next_print += 10

    df = pd.DataFrame(dataset)

    return df

def run_feature_trial(graph_gen_func, seeding_func, n_seeds, algorithm):
    """
    Runs one trial and returns features + match ratio.
    """
    # Generate graphs
    G1, G2_shuffled, optimal_perm = graph_gen_func()

    # Generate seeds
    seeds_G1, seeds_G2 = seeding_func(
        G1,
        G2_shuffled,
        n_seeds,
        optimal_perm
    )

    partial_match = np.column_stack(
        (seeds_G1, seeds_G2)
    )

    # Run matcher
    predicted_perm = algorithm(
        G1,
        G2_shuffled,
        partial_match
    )

    score = np.mean(
        predicted_perm == optimal_perm
    )

    G1_nx = nx.from_numpy_array(G1)
    G2_nx = nx.from_numpy_array(G2_shuffled)
    #extract features
    features = compute_seed_features(
        G1_nx,
        G2_nx,
        seeds_G1,
        seeds_G2
    )
    features["num_seeds"] = n_seeds
    features["match_ratio"] = score

    return features

def run_feature_trial_wrapper(job):
    return run_feature_trial(
        job["graph_gen_func"],
        job["seeding_func"],
        job["seed_count"],
        job["algorithm"],
    )

def compute_seed_features(G1, G2, seeds_G1, seeds_G2):
    """
    Computes features describing a pair of seed sets across two graphs.
    """

    features = {}

    features_G1 = {}
    features_G1.update(compute_graph_features(G1))
    features_G1.update(compute_single_graph_seed_features(G1, seeds_G1))

    features_G2 = {}
    features_G2.update(compute_graph_features(G2))
    features_G2.update(compute_single_graph_seed_features(G2, seeds_G2))

    for key in features_G1:

        value_G1 = features_G1[key]
        value_G2 = features_G2[key]

        features[f"G1_{key}"] = value_G1
        features[f"G2_{key}"] = value_G2
        features[f"{key}_difference"] = abs(value_G1 - value_G2)

    return features

def compute_graph_features(G):
    """
    Computes structural features of the entire graph.
    """

    degrees = np.array([
        degree
        for _, degree in G.degree()
    ])

    return {
        "avg_deg": degrees.mean(),
        "max_deg": degrees.max(),
        "min_deg": degrees.min(),
        "deg_std": degrees.std(),

        "density": nx.density(G),

        "avg_clustering": nx.average_clustering(G),

        "num_connected_components":
            nx.number_connected_components(G),
    }


def compute_single_graph_seed_features(G, seeds):
    """
    Computes structural properties of a seed set inside one graph.
    """

    features = {}

    features.update(compute_degree_features(G, seeds))
    features.update(compute_neighbor_features(G, seeds))
    features.update(compute_clustering_features(G, seeds))
    features.update(compute_distance_features(G, seeds))
    features.update(compute_exposure_features(G, seeds))

    return features


def compute_degree_features(G, seeds):
    """
    Degree statistics of seed nodes.
    """

    degrees = np.array([
        G.degree(node)
        for node in seeds
    ])

    return {
        "avg_seed_deg": degrees.mean(),
        "max_seed_deg": degrees.max(),
        "min_seed_deg": degrees.min(),
        "seed_deg_std": degrees.std(),
    }

def compute_neighbor_features(G, seeds):
    """
    Features describing seed neighborhoods.
    """

    seed_set = set(seeds)

    neighbor_degrees = []
    two_hop_sizes = []

    for seed in seeds:

        neighbors = list(G.neighbors(seed))

        if neighbors:
            neighbor_degrees.append(
                np.mean([
                    G.degree(node)
                    for node in neighbors
                ])
            )

        two_hop = set()

        for neighbor in neighbors:
            two_hop.update(G.neighbors(neighbor))

        two_hop -= seed_set

        two_hop_sizes.append(len(two_hop))

    return {
        "avg_seed_neighbor_deg": (
            np.mean(neighbor_degrees)
            if neighbor_degrees else 0
        ),

        "avg_seed_two_hop_size": (
            np.mean(two_hop_sizes)
            if two_hop_sizes else 0
        ),
    }


def compute_clustering_features(G, seeds):
    """
    Local clustering coefficient statistics of the seed nodes.
    """

    clustering = np.array([
        nx.clustering(G, node)
        for node in seeds
    ])

    return {
        "avg_seed_clustering": clustering.mean(),
        "seed_clustering_std": clustering.std(),
    }


def compute_distance_features(G, seeds):
    """
    Pairwise shortest path statistics between seeds.
    """

    distances = []

    seeds = list(seeds)

    for i in range(len(seeds)):
        for j in range(i + 1, len(seeds)):

            try:
                distance = nx.shortest_path_length(
                    G,
                    seeds[i],
                    seeds[j]
                )

                distances.append(distance)

            except nx.NetworkXNoPath:
                continue

    if len(distances) == 0:
        return {
            "avg_seed_pairwise_dist": 0,
            "max_seed_pairwise_dist": 0,
            "min_seed_pairwise_dist": 0,
        }

    distances = np.array(distances)

    return {
        "avg_seed_pairwise_dist": distances.mean(),
        "max_seed_pairwise_dist": distances.max(),
        "min_seed_pairwise_dist": distances.min(),
    }

def compute_exposure_features(G, seeds):
    """
    Measures how much of the graph is immediately exposed
    by the seed set.
    """

    seed_set = set(seeds)

    exposed_nodes = set()

    for seed in seeds:
        exposed_nodes.update(G.neighbors(seed))

    exposed_nodes -= seed_set

    return {
        "seed_exposure": (
            len(exposed_nodes) /
            G.number_of_nodes()
        )
    }